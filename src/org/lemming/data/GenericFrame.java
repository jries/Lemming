package org.lemming.data;

/**
 * A GenericFrame is an implementation of Frame which includes basic fields like width, height and frame number. It provides a template for a basic Frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public class GenericFrame<T> implements Frame<T> {
	
	T pixels;
	int width;
	int height;
	long frameNo;

	public GenericFrame(long frameNo, int width, int height, T pixels) {
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
	public T getPixels() {
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
