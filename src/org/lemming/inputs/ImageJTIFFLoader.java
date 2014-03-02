package org.lemming.inputs;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import org.lemming.data.ImgLib2Frame;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class ImageJTIFFLoader<T extends RealType<T> & NativeType<T>> extends ImageJWindowLoader<T> {

	String filename; 
	
	@Override
	public void beforeRun() {
		img = new ImagePlus(filename);
	}
	
	public ImageJTIFFLoader(String path) {
		this.filename = path;
	}
}
