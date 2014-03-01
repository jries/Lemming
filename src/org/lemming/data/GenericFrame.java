package org.lemming.data;

import net.imglib2.RandomAccessibleInterval;

/**
 * A GenericFrame is an implementation of Frame which includes basic fields like width, height and frame number. It provides a template for a basic Frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public class GenericFrame<T> implements Frame<T> {
	
	RandomAccessibleInterval<T> pixels;
	int width;
	int height;
	long frameNo;

	public GenericFrame(long frameNo, int width, int height, RandomAccessibleInterval<T> pixels) {
		this.width = width;
		this.height = height;
		this.pixels = pixels;
		this.frameNo = frameNo;
	}

	@Override
	public long getFrameNumber() {
		return frameNo;
	}

	@Override
	public RandomAccessibleInterval<T> getPixels() {
		return pixels;
	}

	@Override
	public int getWidth() {
		return width;
	}

	@Override
	public int getHeight() {
		return height;
	}

}
