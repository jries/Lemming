package org.lemming.modules;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.tools.LemmingUtils;

/**
 * loading images onto the queue
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public class ImageLoader<T extends RealType<T> & NativeType<T>> extends SingleRunModule{
	
	private int curSlice = 0;
	private ImageStack img;
	private int stackSize;
	private double pixelDepth;
	private Double offset;
	private Double em_gain;
	private Double conversion;

	public ImageLoader(ImagePlus loc_im, List<Double> cameraSettings) {
		this.img = loc_im.getStack();
		stackSize = loc_im.getNSlices()*loc_im.getNFrames()*loc_im.getNChannels();
		pixelDepth = loc_im.getCalibration().pixelDepth == 0 ? cameraSettings.get(3) : loc_im.getCalibration().pixelDepth;
		offset = cameraSettings.get(0);
		em_gain = cameraSettings.get(1);
		conversion = cameraSettings.get(2);
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
	}

	@Override
	public Element processData(Element data) {		
		ImageProcessor ip = img.getProcessor(++curSlice);
		Img<T> theImage = LemmingUtils.wrap(ip.getPixels(), new long[]{img.getWidth(), img.getHeight()});

		final Cursor<T> it = theImage.cursor();
		while(it.hasNext()){
			it.fwd();
			final double adu = Math.max((it.get().getRealDouble()-offset), 0);
			final double im2phot = adu*conversion/em_gain;
			it.get().setReal(im2phot);
		}
		ImgLib2Frame<T> frame = new ImgLib2Frame<T>(curSlice, img.getWidth(), img.getHeight(), pixelDepth, theImage);

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
	
	@Override
	public boolean check() {
		return outputs.size()>=1;
	}
}
