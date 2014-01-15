package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.TIFFLoader;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class TIFFLoaderTest {

	Frame f;
	TIFFLoader tif;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new TIFFLoader(p.getProperty("samples.dir")+"eye.tif");
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
