package org.lemming.tools;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;

import java.awt.image.IndexColorModel;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.Localization;

import ij.process.FloatPolygon;

public class LemmingUtils {

	public static FloatPolygon convertToPoints(List<Element> me, double pixelSize) {
		FloatPolygon polygon = new FloatPolygon();
		for (Element el : me) {
			Localization loc = (Localization) el;
			polygon.addPoint(loc.getX() / pixelSize, loc.getY() / pixelSize);
		}
		return polygon;
	}

	@SuppressWarnings("unchecked")
	public static <T extends NativeType<T>> Img<T> wrap(Object ip, long[] dims) {
		
		String className = ip.getClass().getName();

		Img<T> theImage = null;
		if (className.contains("[S")) {
			theImage = (Img<T>) ArrayImgs.unsignedShorts((short[]) ip, dims);
		} else if (className.contains("[F")) {
			theImage = (Img<T>) ArrayImgs.floats((float[]) ip, dims);
		} else if (className.contains("[B")) {
			theImage = (Img<T>) ArrayImgs.unsignedBytes((byte[]) ip, dims);
		} else if (className.contains("[I")) {
			theImage = (Img<T>) ArrayImgs.unsignedInts((int[]) ip, dims);
		} else if (className.contains("[D")) {
			theImage = (Img<T>) ArrayImgs.doubles((double[]) ip, dims);
		}
		return theImage;
	}

	public static IndexColorModel getDefaultColorModel() {
		byte[] r = new byte[256];
		byte[] g = new byte[256];
		byte[] b = new byte[256];
		for (byte i = -128; i < 128; i++) {
			r[i] = i;
			g[i] = i;
			b[i] = i;
		}
		return new IndexColorModel(8, 256, r, g, b);
	}

	public static IndexColorModel Fire() {
		byte[] reds = new byte[256]; 
		byte[] greens = new byte[256]; 
		byte[] blues = new byte[256];
		int[] r = {0,0,1,25,49,73,98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
		int[] g = {0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,57,79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255};
		int[] b = {0,61,96,130,165,192,220,227,210,181,151,122,93,64,35,5,0,0,0,0,0,0,0,0,0,0,0,35,98,160,223,255};
		for (int i=0; i<r.length; i++) {
			reds[i] = (byte)r[i];
			greens[i] = (byte)g[i];
			blues[i] = (byte)b[i];
		}
		return new IndexColorModel(8, 256, reds, greens, blues);
	}


}
