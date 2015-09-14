package org.lemming.modules;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.MultiRunModule;

public class DummyImageLoader<T extends NumericType<T> & NativeType<T>> extends MultiRunModule{
	
	private int curSlice = 0;
	private ImagePlus img;
	private final int stackSize, width, height;
	private long start;
	
	public DummyImageLoader(int stackSize, int width, int height) {
		this.stackSize = stackSize;
		this.width = width;
		this.height = height;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();

		//ImageStack ims = new ImageStack(width, height);
		//img = new ImagePlus("Dummy Dataset");
	}

	@SuppressWarnings({ "unchecked" })
	@Override
	public Element process(Element data) {
		
		if (curSlice >= stackSize){ cancel(); return null; }

		ImageProcessor ip = new FloatProcessor(width, height);
		
		long[] dims = new long[]{ip.getWidth(), ip.getHeight()};
		
		Img<T> theImage = null;
		if (ip instanceof ShortProcessor) {
			theImage = (Img<T>) ArrayImgs.unsignedShorts((short[]) ip.getPixels(), dims);
		} else if (ip instanceof FloatProcessor) {
			theImage = (Img<T>) ArrayImgs.floats((float[])ip.getPixels(), dims);
		} else if (ip instanceof ByteProcessor) {
			theImage = (Img<T>) ArrayImgs.unsignedBytes((byte[])ip.getPixels(), dims);
		}
		
		ImgLib2Frame<T> frame = new ImgLib2Frame<>(curSlice, ip.getWidth(), ip.getHeight(), theImage);
		if (curSlice >= stackSize)
			frame.setLast(true);
		return frame;
	}
	
	@Override
	public void afterRun(){
		System.out.println("Loading done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	
	public void show(){
		img.show();
	}

	@Override
	public boolean check() {
		return outputs.size()>=1;
	}
}
