package org.lemming.modules;

import ij.ImagePlus;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.tools.LemmingUtils;

public class ImageLoader<T extends NumericType<T> & NativeType<T>> extends SingleRunModule{
	
	private int curSlice = 0;
	private ImagePlus img;
	private int stackSize;
	private double pixelDepth;
	
	public ImageLoader(ImagePlus img) {
		this.img = img;
		stackSize = img.getStackSize();
		pixelDepth = img.getCalibration().pixelDepth == 0 ? 1 : img.getCalibration().pixelDepth;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
	}

	@Override
	public Element processData(Element data) {		
		Object ip = img.getImageStack().getPixels(++curSlice);
		
		Img<T> theImage = LemmingUtils.wrap(ip, new long[]{img.getWidth(), img.getHeight()});
		ImgLib2Frame<T> frame = new ImgLib2Frame<>(curSlice, img.getWidth(), img.getHeight(), pixelDepth, theImage);
		if (curSlice >= stackSize){
			frame.setLast(true);
			cancel(); 
			return frame; 
		}
		return frame;
	}
	
	@Override
	public void afterRun(){
		System.out.println("Loading of " + stackSize +" done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	
	public void show(){
		img.show();
	}

	@Override
	public boolean check() {
		return outputs.size()>=1;
	}
}
