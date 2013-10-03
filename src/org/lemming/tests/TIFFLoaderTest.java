package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.input.TIFFLoader;

public class TIFFLoaderTest {

	Frame f;
	TIFFLoader tif;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		tif = new TIFFLoader("tifSample.tif");
		frames = new QueueStore<Frame>();
		
		tif.setOutput(frames);
	}

	@Test
	public void test() {
		tif.run();
		
		assertEquals(600, frames.getLength());
		
		int cnt = 0;
		float[] pixels;
		while (!frames.isEmpty()){
			f = frames.get();
			pixels = (float[])f.getPixels();
			assertEquals(6400, pixels.length);
			System.out.println("Frame: "+ ++cnt + ", Pixel(0,0): " + pixels[0]);
		}
		
		tif.show();
		while(true){}
		
	}

}
