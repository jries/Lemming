package org.lemming.data;

import org.lemming.interfaces.Frame;

import net.imglib2.RandomAccessibleInterval;

/**
 * A GenericFrame is an implementation of Frame which includes basic fields like width, height and frame number. It provides a template for a basic Frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T> - data type
 */
public class GenericFrame<T> implements Frame<T> {
	
	private RandomAccessibleInterval<T> pixels;
	private int width;
	private int height;
	private long frameNo;
	private boolean isLast = false;

	/**
	 * @param frameNo - frame number
	 * @param width - width
	 * @param height - height
	 * @param pixels - pixels
	 */
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

	@Override
	public boolean isLast() {
		return isLast;
	}

	/**
	 * @param isLast - switch
	 */
	public void setLast(boolean isLast) {
		this.isLast = isLast;		
	}

}
