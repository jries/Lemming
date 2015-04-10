package org.lemming.inputs;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import org.lemming.data.ImgLib2Frame;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public class ImageJTIFFLoader<T extends RealType<T> & NativeType<T>> extends SingleOutput<ImgLib2Frame<T>> {

	private int curSlice = 0;
	private String filename; 
	
	private ImagePlus img;
	
	@Override
	public void beforeRun() {
		img = new ImagePlus(filename);
	}
	
	/**
	 * @param path - file path
	 */
	public ImageJTIFFLoader(String path) {
		this.filename = path;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return curSlice < img.getStack().getSize();
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
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
		
		ImgLib2Frame frame = new ImgLib2Frame(curSlice, (int)dims[0], (int)dims[1], theImage);
		if (!hasMoreOutputs())
			frame.setLast(true);
		
		return frame;
	}
	
	@Override
	public void afterRun(){
		System.out.println("loading finished");
	}

	/**
	 * 
	 */
	public void show() {
		img.show();
	}

}
