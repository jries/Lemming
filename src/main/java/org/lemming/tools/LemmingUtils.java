package org.lemming.tools;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;

import ij.process.ByteProcessor;
import ij.process.FloatPolygon;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;


public class LemmingUtils {
	
	public static FloatPolygon convertToPoints(FrameElements me){
		FloatPolygon polygon = new FloatPolygon();
		for (Element el: me.getList()){
			Localization loc = (Localization) el;
			polygon.addPoint(loc.getX(),loc.getY());
		}
		return polygon;
	}
	
	@SuppressWarnings("unchecked")
	public static  <T extends NativeType<T>> Img<T> wrap(ImageProcessor ip){
		long[] dims = new long[]{ip.getWidth(), ip.getHeight()};
		
		Img<T> theImage = null;
		if (ip instanceof ShortProcessor) {
			theImage = (Img<T>) ArrayImgs.unsignedShorts((short[]) ip.getPixels(), dims);
		} else if (ip instanceof FloatProcessor) {
			theImage = (Img<T>) ArrayImgs.floats((float[])ip.getPixels(), dims);
		} else if (ip instanceof ByteProcessor) {
			theImage = (Img<T>) ArrayImgs.unsignedBytes((byte[])ip.getPixels(), dims);
		}
		return theImage;
	}

}
