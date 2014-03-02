package org.lemming.data;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * This class wraps an ImgLib2 image and provides the interface to Frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class ImgLib2Frame <T extends RealType<T>> implements Frame<T> {
	
	long frameNo;
	int width;
	int height;
	RandomAccessibleInterval<T> slice;
	
	/**
	 * Creates a Frame with a reference to the appropriate (2D, although not enforced, yet) frame. 
	 * 
	 * @param frameNo
	 * @param width
	 * @param height
	 * @param slice
	 */
	public ImgLib2Frame(long frameNo, int width, int height, RandomAccessibleInterval<T> slice) {
		this.frameNo = frameNo;
		this.width = width;
		this.height = height;
		this.slice = slice;
	}
	
	@Override
	public long getFrameNumber() {
		return frameNo;
	}

	@Override
	public RandomAccessibleInterval<T> getPixels() {
		return slice;
		//return ImageJFunctions.wrap(slice,"Slice "+frameNo).getProcessor().getPixels();
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
