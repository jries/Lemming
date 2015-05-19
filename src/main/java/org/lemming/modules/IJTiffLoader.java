package org.lemming.modules;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import java.io.File;
import java.util.Map;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.NumericType;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.Module;

public class IJTiffLoader<T extends NumericType<T>> extends Module{
	
	private int curSlice = 0;
	private String filename; 
	private ImagePlus img;
	private int stackSize;
	private String outputkey;
	private long start;
	
	public IJTiffLoader(String path) {
		filename = path;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		if (new File(filename).exists()){
			img = new ImagePlus(filename);
			stackSize = img.getStack().getSize();
		}
		else 
			System.err.println("File not exist!");
		
		outputkey = outputs.keySet().iterator().next(); // for this module there should be only one key
	}

	public boolean hasMoreOutputs() {
		return curSlice < stackSize;
	}

	@SuppressWarnings({ "unchecked" })
	@Override
	public void process(Map<String, Element> data) {
		
		if (curSlice >= stackSize){ cancel(); return; }
		
		ImageProcessor ip = img.getStack().getProcessor(++curSlice);
		
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
		if (!hasMoreOutputs())
			frame.setLast(true);
		data.put(outputkey, frame);
	}
	
	@Override
	public void afterRun(){
		System.out.println("Loading done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	
	public void show(){
		img.show();
	}

}
