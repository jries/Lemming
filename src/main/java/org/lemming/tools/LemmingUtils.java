package org.lemming.tools;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;

import java.awt.image.IndexColorModel;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.Localization;

import ij.process.FloatPolygon;

/**
 * General utility class
 * 
 * @author Ronny Sczech
 *
 */
public class LemmingUtils {

	public static FloatPolygon convertToPoints(List<Element> me, float pixelSize) {
		FloatPolygon polygon = new FloatPolygon();
		for (Element el : me) {
			Localization loc = (Localization) el;
			polygon.addPoint(loc.getX().floatValue() / pixelSize, loc.getY().floatValue() / pixelSize);
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

	public static IndexColorModel Ice() {
		byte[] reds = new byte[256]; 
		byte[] greens = new byte[256]; 
		byte[] blues = new byte[256];
        int[] r = {0,0,0,0,0,0,0,19,29,50,48,79,112,134,158,186,201,217,229,242,250,250,250,250,251,250,250,250,250,251,251,243,230};
        int[] g = {0,156,165,176,184,190,196,193,184,171,162,146,125,107,93,81,87,92,97,95,93,93,90,85,69,64,54,47,35,19,0,4,0};
        int[] b = {0,140,147,158,166,170,176,209,220,234,225,236,246,250,251,250,250,245,230,230,222,202,180,163,142,123,114,106,94,84,64,26,27};
		for (int i=0; i<r.length; i++) {
			reds[i] = (byte)r[i];
			greens[i] = (byte)g[i];
			blues[i] = (byte)b[i];
		}
		return new IndexColorModel(8, 256, reds, greens, blues);
	}


}
