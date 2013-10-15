package org.lemming.data;

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
