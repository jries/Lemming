package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.math.Gradient;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.img.Img;
import net.imglib2.view.Views;

public class GradientFitterTest {

	private double[] result;

	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Before
	public void setUp() {
		final String filename = System.getProperty("user.home")+"/ownCloud/bead.tif";
		ImagePlus img = new ImagePlus(filename);
		Object ip = img.getStack().getPixels(1);
		
		int width = img.getWidth();
		int height = img.getHeight();
		Img slice = LemmingUtils.wrap(ip, new long[]{width, height});
		FinalInterval interval = new FinalInterval(new long[]{26,25},new long[]{40,39});
		Gradient gf = new Gradient(Views.interval(slice, interval), 0, 3);
		result = gf.fit();
	}

	@Test
	public void test() {
		assertEquals("ellipticity OK",result[2],0.608710038707477,1e-2);
	}

}
