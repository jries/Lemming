package org.lemming.plugins;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.lemming.factories.PreProcessingFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.FastMedianPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.QuickSelect;
import org.lemming.modules.ImageMath.operators;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;
import org.scijava.plugin.Plugin;

import javolution.util.FastTable;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class FastMedianFilter<T extends IntegerType<T> & NativeType<T>> extends SingleRunModule {
	
	public static final String NAME = "Fast Median Filter";

	public static final String KEY = "FASTMEDIAN";

	public static final String INFO_TEXT = "<html>" + "Fast Median Filter using a 3x3 kernel times the given frames for calculating an approximate median."
			+ "It has an option to interpolate between blocks" + "</html>";

	private int nFrames;

	private boolean interpolating;
	
	private Queue<Frame<T>> frameList = new ArrayDeque<>();

	private int counter = 0;

	private int lastListSize = 0;

	private Frame<T> frameA = null;
	
	private Frame<T> frameB = null;

	public FastMedianFilter(final int numFrames, boolean interpolating) {
		this.nFrames = numFrames;
		this.interpolating = interpolating;
	}

	@Override
	public boolean check() {
		return inputs.size() == 1 && outputs.size() >= 1;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		final Frame<T> frame = (Frame<T>) data;
		
		frameList.add(frame);
		counter++;
		
		if (frame.isLast()) {// process the rest;
			Queue<Frame<T>> transferList = new ArrayDeque<>();
			transferList.addAll(frameList);
			frameB = process(transferList, true);
			if(interpolating)
				interpolate();
			running = false;
			lastListSize = frameList.size() - 1;
			lastFrames();
			frameList.clear();
			return null;
		}
		
		if (counter % nFrames == 0) {// make a new list for each Callable
			Queue<Frame<T>> transferList = new ArrayDeque<>();
			transferList.addAll(frameList);
			frameB = process(transferList, false);
			if(interpolating){
				if (frameA != null){
					interpolate();
				}
				frameA = frameB;
			} else {
				for (int i = 0; i < nFrames; i++)
					newOutput(new ImgLib2Frame<>(frameB.getFrameNumber()
							+ i, frameB.getWidth(), frameB.getWidth(), frameB.getPixelDepth(), 
							frameB.getPixels()));				
			}
			frameList.clear();
		}
		return null;
	}
	
	private Frame<T> process(final Queue<Frame<T>> list, final boolean isLast) {
		Frame<T> newFrame = null;
		if (!list.isEmpty()){
			final Frame<T> firstFrame = list.peek();
			final RandomAccessibleInterval<T> firstInterval = firstFrame.getPixels();	// handle borders
			final long[] dims = new long[firstInterval.numDimensions()];
			firstInterval.dimensions(dims);
			Img<T> out = new ArrayImgFactory<T>().create(dims, Views
					.iterable(firstInterval).firstElement());
			
			final FinalInterval shrinked = Intervals.expand(out,-1);		// handle borders
			final IntervalView<T> source = Views.interval( out, shrinked );
			final RectangleShape outshape = new RectangleShape(1, false);	// 3x3 kernel
			Cursor<T> outcursor = Views.iterable(source).cursor();
			
			List<RandomAccess<T>> cursorList = new FastTable<RandomAccess<T>>();
			
			for (Frame<T> currentFrame : list){
				RandomAccessibleInterval<T> currentInterval = currentFrame.getPixels();
				cursorList.add(currentInterval.randomAccess());								// creating neighborhoods
			}
			
			for ( final Neighborhood< T > localNeighborhood : outshape.neighborhoods( source ) )
	        {
				outcursor.fwd();
				final Cursor<T> localCursor = localNeighborhood.cursor();
				final List<Integer> values = new FastTable<Integer>();
				while(localCursor.hasNext()){
					localCursor.fwd();
					for (RandomAccess<T> currentCursor : cursorList) {
						currentCursor.setPosition(localCursor);
						values.add(currentCursor.get().getInteger());
					}
				}
				final Integer median = QuickSelect.fastmedian(values, values.size());   // find the median
				if (median != null)
					outcursor.get().setInteger(median);
	        }
			
			// Borders
			final Cursor<T>  top = Views.interval(out, Intervals.createMinMax(0,0,dims[0]-1,0)).cursor();
			while(top.hasNext()){
				final List<Integer> values = new FastTable<Integer>();
				top.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(top);
					values.add(currentCursor.get().getInteger());
				}
				final Integer median = QuickSelect.fastmedian(values, values.size());   // find the median
				if (median != null)
					top.get().setInteger(median);
			}
			final Cursor<T>  left = Views.interval(out,Intervals.createMinMax(0,1,0,dims[1]-2)).cursor();
			while(left.hasNext()){
				final List<Integer> values = new FastTable<Integer>();
				left.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(left);
					values.add(currentCursor.get().getInteger());
				}
				final Integer median = QuickSelect.fastmedian(values, values.size());   // find the median
				if (median != null)
					left.get().setInteger(median);
			}
			final Cursor<T>  right = Views.interval(out,Intervals.createMinMax(dims[0]-1,1,dims[0]-1,dims[1]-2)).cursor();
			while(right.hasNext()){
				final List<Integer> values = new FastTable<Integer>();
				right.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(right);
					values.add(currentCursor.get().getInteger());
				}
				final Integer median = QuickSelect.fastmedian(values, values.size());   // find the median
				if (median != null)
					right.get().setInteger(median);
			}
			final Cursor<T>  bottom = Views.interval(out,Intervals.createMinMax(0,dims[1]-1,dims[0]-1,dims[1]-1)).cursor();
			while(bottom.hasNext()){
				final List<Integer> values = new FastTable<Integer>();
				bottom.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(bottom);
					values.add(currentCursor.get().getInteger());
				}
				final Integer median = QuickSelect.fastmedian(values, values.size());   // find the median
				if (median != null)
					bottom.get().setInteger(median);
			}		
			
			newFrame = new ImgLib2Frame<T>(firstFrame.getFrameNumber(), firstFrame.getWidth(), 
			firstFrame.getHeight(), firstFrame.getPixelDepth(), out);
		} else {
			newFrame = new ImgLib2Frame<T>(0, 1, 1, 1, null);
		}
		if (isLast)
			newFrame.setLast(true);
		return newFrame;
	}
	
	/**
	 * interpolate between blocks
	 */
	private void interpolate(){
		RandomAccessibleInterval<T> intervalA = frameA.getPixels();
		RandomAccessibleInterval<T> intervalB = frameB.getPixels();

		for (int i = 0; i < nFrames; i++) {
			Img<T> outFrame = new ArrayImgFactory<T>()
					.create(intervalA, Views.iterable(intervalA)
							.firstElement());
			Cursor<T> outCursor = outFrame.cursor();
			Cursor<T> cursorA = Views.iterable(intervalA).cursor();
			Cursor<T> cursorB = Views.iterable(intervalB).cursor();

			while (cursorA.hasNext()) {
				cursorA.fwd();
				cursorB.fwd();
				outCursor.fwd();
				outCursor.get().setInteger(
					cursorA.get().getInteger()
					+ Math.round((cursorB.get().getInteger() - cursorA.get().getInteger()) * ((float) i + 1) / nFrames));
			}

			newOutput(new ImgLib2Frame<>(
					frameA.getFrameNumber() + i, frameA.getWidth(),
					frameA.getHeight(), frameA.getPixelDepth(), outFrame));
		}
	}
	
	private void lastFrames(){
		// handle the last frames
		for (int i = 0; i < lastListSize; i++) {
			newOutput(new ImgLib2Frame<>(frameB.getFrameNumber() + i,
					frameB.getWidth(), frameB.getHeight(), 
					frameB.getPixelDepth(), frameB.getPixels()));
		}

		// create last frame
		ImgLib2Frame<T> lastFrame = new ImgLib2Frame<>(
				frameB.getFrameNumber() + lastListSize, frameB.getWidth(),
				frameB.getHeight(), frameB.getPixelDepth(), frameB.getPixels());
		lastFrame.setLast(true);
		newOutput(lastFrame);
	}
	
	@Override
	public void afterRun(){
		System.out.println("Filtering done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	
	/**
	 * Factory for the Fast Median Filter implementation
	 * 
	 * @author Ronny Sczech
	 *
	 */
	
	@Plugin(type = PreProcessingFactory.class, visible = true)
	public static class Factory implements PreProcessingFactory {

		private Map<String, Object> settings;
		private FastMedianPanel configPanel = new FastMedianPanel();

		@Override
		public String getInfoText() {
			return INFO_TEXT;
		}

		@Override
		public String getKey() {
			return KEY;
		}

		@Override
		public String getName() {
			return NAME;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return true;
		}

		@SuppressWarnings("rawtypes")
		@Override
		public AbstractModule getModule() {
			boolean interpolating = (boolean) settings.get(FastMedianPanel.KEY_INTERPOLATING);
			int frames = (int) settings.get(FastMedianPanel.KEY_FRAMES);
			return new FastMedianFilter(frames, interpolating);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public operators getOperator() {
			return operators.SUBSTRACTION;
		}

		@Override
		public int processingFrames() {
			int procFrames = (Integer) settings.get(FastMedianPanel.KEY_FRAMES) == 0 ? 1 : (Integer) settings.get(FastMedianPanel.KEY_FRAMES);
			return procFrames;
		}

	}

}
