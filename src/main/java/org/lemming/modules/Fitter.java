package org.lemming.modules;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.CircularFifoQueue;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.MultiRunModule;

public abstract class Fitter<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {

	private long start;
	protected long size;
	private Store locInput;
	private volatile CircularFifoQueue<Element> firstQueue;
	private Store secondQueue;
	private int queueSize;

	public Fitter(int queueSize, long windowSize) {
		this.size = windowSize;
		this.queueSize = queueSize;
		firstQueue = queueSize == 0 ? new CircularFifoQueue<Element>(128) : new CircularFifoQueue<Element>(queueSize);
		secondQueue = new FastStore();
	}


	@Override
	protected void beforeRun() {
		Iterator<Integer> it = inputs.keySet().iterator();
		try {
			iterator = it.next(); 							// first input
			locInput = inputs.get(it.next()); 				// second input
		} catch (NoSuchElementException | NullPointerException ex){
			System.err.println("Input provided not correct!");
			Thread.currentThread().interrupt();
		}
		
		start = System.currentTimeMillis();
		while (locInput.getLength()<queueSize) pause(10);
		fillQueue();
	}
	
	private void fillQueue(){
		for (int i=firstQueue.size();i<queueSize;i++){
			Element el = locInput.get();
			if (el instanceof FrameElements) 
				firstQueue.add(el);		
		}
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {
		F frame = (F) data;
		if (frame == null)
			return null;

		if (frame.isLast()) {
			if (!inputs.get(iterator).isEmpty()){
				inputs.get(iterator).put(frame);
				return null;
			}
			process1(frame);
			cancel();
		}
		
		process1(frame);
		return null;
	}

	private void process1(F frame) {
		List<Element> sliceLocs = new ArrayList<>();
		
		if (firstQueue.isEmpty()) fillQueue();
		for (Element element : firstQueue){
			if (element instanceof FrameElements){
				FrameElements fe = (FrameElements) element;
				if (fe.getNumber()==frame.getFrameNumber()){
					for ( Element l : fe.getList())
						sliceLocs.add(l);
					firstQueue.remove(element);
					Element newEl = locInput.get();
					if (newEl!=null) 
						firstQueue.add(newEl);		
				}
			}
		}
		
		if (sliceLocs.isEmpty()){
			secondQueue.put(frame);
			return;
		}	
		
		fit( sliceLocs, frame.getPixels(), size);
	}
	
	public abstract void fit(List<Element> sliceLocs, RandomAccessibleInterval<T> pixels, long windowSize);
		

	@SuppressWarnings({ "unchecked" })
	@Override
	protected void afterRun() {
		Store finisherList = new FastStore(); 
		if(!firstQueue.isEmpty())
			for (Element q : firstQueue)
				finisherList.put(q);
		firstQueue.clear();
		while (!locInput.isEmpty()){
			Element q = locInput.get();
			if (q != null)
				finisherList.put(q);
		}
		Collection<Element> view = finisherList.view();

		List<Element> sliceLocs = new ArrayList<>();
		
		int secondRun = secondQueue.getLength();
		
		while (!secondQueue.isEmpty()){
			F frame = (F) secondQueue.get();
			for (Element element : view){
				if (element instanceof FrameElements){
					FrameElements fe = (FrameElements) element;
					if (fe.getNumber()==frame.getFrameNumber()){
						for ( Element l : fe.getList())
							sliceLocs.add(l);
					}
				}
			}
			if (!sliceLocs.isEmpty()){
				fit(sliceLocs,frame.getPixels(), size);
				sliceLocs.clear();
			}
		}
		FittedLocalization lastLoc = new FittedLocalization(0, -1, -1, 0, -1, -1) ;
		lastLoc.setLast(true);
		newOutput(lastLoc);
		System.out.println("Fitting done in " + (System.currentTimeMillis() - start) + "ms with elements in 2nd run:" + secondRun);
	}

	@Override
	public boolean check() {
		return inputs.size()==2 && outputs.size()>=1;
	}

}
