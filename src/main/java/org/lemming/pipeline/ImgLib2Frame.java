package org.lemming.pipeline;

import org.lemming.interfaces.Frame;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.NumericType;

public class ImgLib2Frame<T extends NumericType<T>> implements Frame<T> {

	final private long frameNo;
	final private int width;
	final private int height;
	final private RandomAccessibleInterval<T> slice;
	private boolean isLast = false;
    final private double pixelDepth;

	/**
	 * Creates a Frame with a reference to the appropriate (2D, although not
	 * enforced, yet) frame.
	 * 
	 * @param frameNo
	 *            - frame number
	 * @param width
	 *            - width
	 * @param height
	 *            - height
	 * @param pixelDepth
	 *            - pixelDepth
	 * @param slice
	 *            - slice
	 */
	public ImgLib2Frame(long frameNo, int width, int height, double pixelDepth, RandomAccessibleInterval<T> slice) {
		this.frameNo = frameNo;
		this.width = width;
		this.height = height;
		this.slice = slice;
		this.pixelDepth = pixelDepth;
	}

	@Override
	public long getFrameNumber() {
		return frameNo;
	}

	@Override
	public RandomAccessibleInterval<T> getPixels() {
		return slice;
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
	
	@Override
	public double getPixelDepth(){
		return pixelDepth;
	}

	/**
	 * @param isLast
	 *            - switch
	 */
	public void setLast(boolean isLast) {
		this.isLast = isLast;
	}

	@Override
	public int compareTo(Frame<T> o) {
		if (this.getFrameNumber() > o.getFrameNumber())
			return 1;
		else if (this.getFrameNumber() == o.getFrameNumber()) return 0;
		else return -1;
	}
	
	@Override
	public String toString(){
		return "F_"+getFrameNumber();
	}
	
}
