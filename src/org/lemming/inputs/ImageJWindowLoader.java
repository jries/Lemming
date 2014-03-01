package org.lemming.inputs;

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import org.lemming.data.ImgLib2Frame;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class ImageJWindowLoader<T extends RealType<T> & NativeType<T>> extends SO<ImgLib2Frame<T>> {

	int curSlice = 0;
	
	ImagePlus img;
	
	@Override
	public void beforeRun() {
		img = WindowManager.getCurrentImage();
	}
	
	public ImageJWindowLoader() {
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return curSlice < img.getStack().getSize();
	}

	@Override
	protected ImgLib2Frame<T> newOutput() {
		curSlice++;
		
		ImageProcessor ip = img.getStack().getProcessor(curSlice);
		
		long[] dims = new long[]{ip.getWidth(), ip.getHeight()};
		
		Img theImage = null;
		if (ip instanceof ShortProcessor) {
			theImage = ArrayImgs.unsignedShorts((short[]) ip.getPixels(), dims);
		} else if (ip instanceof FloatProcessor) {
			theImage = ArrayImgs.floats((float[])ip.getPixels(), dims);
		} else if (ip instanceof ByteProcessor) {
			theImage = ArrayImgs.unsignedBytes((byte[])ip.getPixels(), dims);
		}
		
		return new ImgLib2Frame(curSlice, (int)dims[0], (int)dims[1], theImage);
	}

	public void show() {
		img.show();
	}

}
