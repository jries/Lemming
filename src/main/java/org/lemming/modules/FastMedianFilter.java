package org.lemming.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import javolution.util.FastTable;

import org.lemming.math.QuickSelect;
import org.lemming.pipeline.Element;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.view.Views;

public class FastMedianFilter<T extends IntegerType<T> & NativeType<T>, F extends Frame<T>> extends SingleRunModule {
	
	private String outputKey;
	private String inputKey;
	private int nFrames, counter = 0, bgCounter = 0; 
	private FastTable<F> frameList = new FastTable<>();
	private long start; 
	private FastTable<Callable<F>> callables = new FastTable<>();
	
	public FastMedianFilter(final int numFrames, final String in, final String out){
		inputKey = in;
		outputKey = out;
		nFrames = numFrames;
	}
	
	@Override
	protected void beforeRun(){
		start = System.currentTimeMillis();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) { // interpolate medians to avoid jumps
		final F frame = (F) data.get(inputKey);
		if (frame == null) return;
		
		frameList.add( frame );
		counter++;	
		
		if(frame.isLast()){ //process the rest;
			callables.add(new FrameCallable(frameList, true)); 
			running = false;
			return;
		}
		
		if (counter % nFrames == 0){
			FastTable<F> transferList = new FastTable<>();
			transferList.addAll(frameList);
			callables.add(new FrameCallable(transferList, false));
			frameList.clear();
		}
	}
	
	class FrameCallable implements Callable<F> {

		private FastTable<F> list;
		private boolean isLast;

		public FrameCallable(final FastTable<F> list, final boolean isLast){
			this.list = list;
			this.isLast = isLast;
		}
		
		@Override
		public F call() throws Exception {
			return process1(list, isLast);
		}

	}

	@SuppressWarnings({ "unchecked" })
	private F process1(final FastTable<F> list, final boolean isLast) {
		if (list.isEmpty()) return null;
		final int middle = nFrames / 2; // integer division
		F firstFrame = list.peek();
		RandomAccessibleInterval<T> firstInterval = firstFrame.getPixels();		
		
		Img<T> out = new ArrayImgFactory<T>().create(firstInterval, Views.iterable( firstInterval ).firstElement().createVariable());
		Cursor<T> cursor = Views.iterable( out ).cursor();
		
		while (cursor.hasNext()){
			cursor.fwd();
			FastTable<Integer> values = new FastTable<>();
			
			for (F currentFrame : list){			
				RandomAccess<T> ra = currentFrame.getPixels().randomAccess(); // maybe this is inefficient
				ra.setPosition(cursor);
				values.add(ra.get().getInteger());
			}
			
			Integer median = QuickSelect.select(values, middle); // find the median
			//values.sort();
			cursor.get().setInteger(median);
		}
		bgCounter++;
		F newFrame = (F) new ImgLib2Frame<>(firstFrame.getFrameNumber(), firstFrame.getWidth(), firstFrame.getHeight(), out);
		if (isLast) newFrame.setLast(true);
		return newFrame;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void afterRun(){
		Store<Frame<T>> outStore = outputs.get(outputKey);
		
		List<F> results = new ArrayList<>();
		
		try{
			List<Future<F>> futures = service.invokeAll(callables);
			
			service.shutdown();
			
			for ( final Future<F> f : futures ){
					F val = f.get();
					if (val!=null)
						results.add(val);
				}
		}
		catch ( final InterruptedException | ExecutionException | CancellationException e ){
			System.err.println(e.getMessage());
		}
		
		Collections.sort(results);
		for (F element : results){
			outStore.put(element);
		}
		
		try {
			service.awaitTermination(1, TimeUnit.MINUTES);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Images created:" + bgCounter + " in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
