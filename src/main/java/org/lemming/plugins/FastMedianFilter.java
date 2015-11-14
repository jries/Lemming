package org.lemming.plugins;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import javolution.util.FastTable;

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

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.view.Views;

public class FastMedianFilter<T extends IntegerType<T> & NativeType<T>, F extends Frame<T>> extends SingleRunModule {

	private int nFrames, counter = 0;
	private FastTable<F> frameList = new FastTable<>();
	private FastTable<Callable<F>> callables = new FastTable<>();
	private int lastListSize = 0;
	private boolean interpolating;
	public static final String NAME = "Fast Median Filter";

	public static final String KEY = "FASTMEDIAN";

	public static final String INFO_TEXT = "<html>" + "Fast Median Filter with the option to interpolate between blocks" + "</html>";

	public FastMedianFilter(final int numFrames, boolean interpolating) {
		nFrames = numFrames;
		this.interpolating = interpolating;
	}


	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		final F frame = (F) data;
		if (frame == null)
			return null;

		frameList.add(frame);
		counter++;

		if (frame.isLast()) {// process the rest;
			callables.add(new FrameCallable(frameList, true));
			running = false;
			lastListSize = frameList.size() - 1;
			return null;
		}

		if (counter % nFrames == 0) {// make a new list for each callable
			FastTable<F> transferList = new FastTable<>();
			transferList.addAll(frameList);
			callables.add(new FrameCallable(transferList, false));
			frameList.clear();
		}
		return null;
	}

	private class FrameCallable implements Callable<F> {

		private FastTable<F> list;
		private boolean isLast;

		public FrameCallable(final FastTable<F> list, final boolean isLast) {
			this.list = list;
			this.isLast = isLast;
		}

		@Override
		public F call() throws Exception {
			return process(list, isLast);
		}

	}

	@SuppressWarnings({ "unchecked" })
	private F process(final Queue<F> list, final boolean isLast) {
		if (list.isEmpty())
			return null;

		final F firstFrame = list.peek();
		final RandomAccessibleInterval<T> firstInterval = firstFrame.getPixels();

		Img<T> out = new ArrayImgFactory<T>().create(firstInterval, Views.iterable(firstInterval).firstElement());
		Cursor<T> cursor = Views.iterable(out).cursor();

		List<Cursor<T>> cursorList = new ArrayList<>();

		for (F currentFrame : list)
			cursorList.add(Views.iterable(currentFrame.getPixels()).cursor());

		while (cursor.hasNext()) {
			FastTable<Integer> values = new FastTable<>();

			cursor.fwd();
			for (Cursor<T> currentCursor : cursorList) {
				currentCursor.fwd();
				values.add(currentCursor.get().getInteger());
			}
			// find the median

			Integer median = QuickSelect.fastmedian(values, values.size());
			// Integer median = QuickSelect.select(values, middle);
			if (median != null)
				cursor.get().setInteger(median);
		}
		F newFrame = (F) new ImgLib2Frame<>(firstFrame.getFrameNumber(), firstFrame.getWidth(), firstFrame.getHeight(), 
				firstFrame.getPixelDepth(), out);
		if (isLast)
			newFrame.setLast(true);
		return newFrame;
	}

	@Override
	protected void afterRun() {

		List<F> results = new ArrayList<>();

		try {
			List<Future<F>> futures = service.invokeAll(callables);

			service.shutdown();

			for (final Future<F> f : futures) {
				F val = f.get();
				if (val != null)
					results.add(val);
			}
		} catch (final InterruptedException | ExecutionException | CancellationException e) {
			System.err.println(e.getMessage());
		}

		Collections.sort(results);

		if (interpolating) {
			// Prepare frame pairs in order
			final Iterator<F> frameIterator = results.iterator();
			F frameA = frameIterator.next();
			F frameB = null;

			while (frameIterator.hasNext()) {
				frameB = frameIterator.next();

				RandomAccessibleInterval<T> intervalA = frameA.getPixels();
				RandomAccessibleInterval<T> intervalB = frameB.getPixels();

				for (int i = 0; i < nFrames; i++) {
					Img<T> outFrame = new ArrayImgFactory<T>().create(intervalA, Views.iterable(intervalA).firstElement());
					Cursor<T> outCursor = outFrame.cursor();
					Cursor<T> cursorA = Views.iterable(intervalA).cursor();
					Cursor<T> cursorB = Views.iterable(intervalB).cursor();

					while (cursorA.hasNext()) {
						cursorA.fwd();
						cursorB.fwd();
						outCursor.fwd();
						outCursor.get().setInteger(cursorA.get().getInteger()
								+ Math.round((cursorB.get().getInteger() - cursorA.get().getInteger()) * ((float) i + 1) / nFrames));
					}

					newOutput(
							new ImgLib2Frame<>(frameA.getFrameNumber() + i, frameA.getWidth(), frameA.getHeight(), frameA.getPixelDepth(), outFrame));
				}
				frameA = frameB;
			}
			if (frameB == null)
				return;
			// handle the last frames
			for (int i = 0; i < lastListSize; i++) {
				newOutput(new ImgLib2Frame<>(frameB.getFrameNumber() + i, frameB.getWidth(), frameB.getHeight(), frameB.getPixelDepth(),
						frameB.getPixels()));
			}

			// create last frame
			ImgLib2Frame<T> lastFrame = new ImgLib2Frame<>(frameB.getFrameNumber() + lastListSize, frameB.getWidth(), frameB.getHeight(),
					frameB.getPixelDepth(), frameB.getPixels());
			lastFrame.setLast(true);
			newOutput(lastFrame);
		} else {
			F lastElements = results.remove(results.size() - 1);
			for (F element : results) {
				for (int i = 0; i < nFrames; i++)
					newOutput(new ImgLib2Frame<>(element.getFrameNumber() + i, element.getWidth(), element.getWidth(), element.getPixelDepth(),
							element.getPixels()));
			}
			// handle the last frames
			for (int i = 0; i < lastListSize; i++) {
				newOutput(new ImgLib2Frame<>(lastElements.getFrameNumber() + i, lastElements.getWidth(), lastElements.getHeight(),
						lastElements.getPixelDepth(), lastElements.getPixels()));
			}
			// create last frame
			ImgLib2Frame<T> lastFrame = new ImgLib2Frame<>(lastElements.getFrameNumber() + lastListSize, lastElements.getWidth(),
					lastElements.getHeight(), lastElements.getPixelDepth(), lastElements.getPixels());
			lastFrame.setLast(true);
			newOutput(lastFrame);
		}

		try {
			service.awaitTermination(1, TimeUnit.MINUTES);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Filtering of " + counter + " images done in " + (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size() == 1 && outputs.size() >= 1;
	}

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
