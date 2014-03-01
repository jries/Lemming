package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.inputs.ScifioLoader;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class ImageJTIFFLoaderTest {

	Frame f;
	ImageJTIFFLoader tif;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ImageJTIFFLoader(p.getProperty("samples.dir")+"eye.tif");
		frames = new QueueStore<Frame>();
		
		tif.setOutput(frames);
	}

	@Test
	public void test() {
		tif.run();
		
		assertEquals(41, frames.getLength());		
		
		tif.show();

		LemMING.pause(5000);
	}

}
