package org.lemming.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
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
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.view.Views;

public class FastMedianFilter<T extends IntegerType<T> & NativeType<T>, F extends Frame<T>>
		extends SingleRunModule {

	private String outputKey;
	private String inputKey;
	private int nFrames, counter = 0;
	private FastTable<F> frameList = new FastTable<>();
	private long start;
	private FastTable<Callable<F>> callables = new FastTable<>();
	private int lastListSize = 0;
	private boolean interpolating;

	public FastMedianFilter(final int numFrames, boolean interpolating) {
		nFrames = numFrames;
		this.interpolating = interpolating;
	}

	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
		// for this module there should be only one key
		inputKey = inputs.keySet().iterator().next(); 
		// for this module there should be only one key											
		outputKey = outputs.keySet().iterator().next(); 											
	}

	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		final F frame = (F) data.get(inputKey);
		if (frame == null)
			return;

		frameList.add(frame);
		counter++;

		if (frame.isLast()) {// process the rest;
			callables.add(new FrameCallable(frameList, true));
			running = false;
			lastListSize = frameList.size() - 1;
			return;
		}

		if (counter % nFrames == 0) {// make a new list for each callable
			FastTable<F> transferList = new FastTable<>();
			transferList.addAll(frameList);
			callables.add(new FrameCallable(transferList, false));
			frameList.clear();
		}
	}

	class FrameCallable implements Callable<F> {

		private FastTable<F> list;
		private boolean isLast;

		public FrameCallable(final FastTable<F> list, final boolean isLast) {
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
		if (list.isEmpty())
			return null;
		final int middle = nFrames / 2; // integer division
		F firstFrame = list.peek();
		RandomAccessibleInterval<T> firstInterval = firstFrame.getPixels();

		Img<T> out = new ArrayImgFactory<T>().create(firstInterval, Views
				.iterable(firstInterval).firstElement());
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
			Integer median = QuickSelect.select(values, middle); 												
			// values.sort();
			// Integer media = values.get(middle);
			cursor.get().setInteger(median);
		}
		F newFrame = (F) new ImgLib2Frame<>(firstFrame.getFrameNumber(),
				firstFrame.getWidth(), firstFrame.getHeight(), out);
		if (isLast)
			newFrame.setLast(true);
		return newFrame;
	}

	@SuppressWarnings({ "unchecked", "null" })
	@Override
	protected void afterRun() {
		Store<Frame<T>> outStore = outputs.get(outputKey);

		List<F> results = new ArrayList<>();

		try {
			List<Future<F>> futures = service.invokeAll(callables);

			service.shutdown();

			for (final Future<F> f : futures) {
				F val = f.get();
				if (val != null)
					results.add(val);
			}
		} catch (final InterruptedException | ExecutionException
				| CancellationException e) {
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
										+ Math.round((cursorB.get()
												.getInteger() - cursorA.get()
												.getInteger())
												* ((float) i + 1) / nFrames));
					}

					outStore.put(new ImgLib2Frame<>(
							frameA.getFrameNumber() + i, frameA.getWidth(),
							frameA.getHeight(), outFrame));
				}
				frameA = frameB;
			}

			// handle the last frames
			for (int i = 0; i < lastListSize; i++) {
				outStore.put(new ImgLib2Frame<>(frameB.getFrameNumber() + i,
						frameB.getWidth(), frameB.getHeight(), frameB
								.getPixels()));
			}

			// create last frame
			ImgLib2Frame<T> lastFrame = new ImgLib2Frame<>(
					frameB.getFrameNumber() + lastListSize, frameB.getWidth(),
					frameB.getHeight(), frameB.getPixels());
			lastFrame.setLast(true);
			outStore.put(lastFrame);
		} else {
			F lastElements = results.remove(results.size()-1);
			for (F element : results) {
				for (int i = 0; i < nFrames; i++)
					outStore.put(new ImgLib2Frame<>(element.getFrameNumber()
							+ i, element.getWidth(), element.getWidth(),
							element.getPixels()));
			}
			// handle the last frames
			for (int i = 0; i < lastListSize; i++) {
				outStore.put(new ImgLib2Frame<>(lastElements.getFrameNumber() + i,
						lastElements.getWidth(), lastElements.getHeight(), lastElements
								.getPixels()));
			}
			// create last frame
			ImgLib2Frame<T> lastFrame = new ImgLib2Frame<>(
					lastElements.getFrameNumber() + lastListSize, lastElements.getWidth(),
					lastElements.getHeight(), lastElements.getPixels());
			lastFrame.setLast(true);
			outStore.put(lastFrame);
		}

		try {
			service.awaitTermination(1, TimeUnit.MINUTES);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Filtering of " + counter + " images done in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

}
