package org.lemming.data;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;

public class ImgLib2Frame implements Frame {
	
	long frameNo;
	int width;
	int height;
	RandomAccessibleInterval<FloatType> slice;
	
	public ImgLib2Frame(long frameNo, int width, int height, RandomAccessibleInterval<FloatType> slice) {
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
	public Object getPixels() {
		return ImageJFunctions.wrap(slice,"Slice "+frameNo).getProcessor().getPixels();
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
