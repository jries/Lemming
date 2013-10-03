package org.lemming.data;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;

public class ImgLib2Frame implements Frame {
	
	long frameNo;
	RandomAccessibleInterval<FloatType> slice;
	
	public ImgLib2Frame(long frameNo, RandomAccessibleInterval<FloatType> slice) {
		this.frameNo = frameNo;
		this.slice = slice;
	}
	
	@Override
	public long getFrameNumber() {
		return 0;
	}

	@Override
	public Object getPixels() {
		return ImageJFunctions.wrap(slice,"Slice "+frameNo).getProcessor().getPixels();
	}
	
	
	
}
