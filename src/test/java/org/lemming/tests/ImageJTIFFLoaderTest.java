package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.interfaces.Frame;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
@SuppressWarnings("rawtypes")
public class ImageJTIFFLoaderTest {

	Frame f;
	ImageJTIFFLoader tif;
	QueueStore<Frame> frames;
	private Properties p;
	
	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		p = new Properties();
		p.load(new FileReader("test.properties"));
		tif = new ImageJTIFFLoader(p.getProperty("samples.dir")+"4b_green.tif");
		frames = new QueueStore<Frame>();
		
		tif.setOutput(frames);
	}

	@Test
	public void test() {
		tif.run();
		
		assertEquals(61, frames.getLength());		
		
		tif.show();

		LemMING.pause(5000);
	}

}
