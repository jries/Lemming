package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ScifioLoader;
import org.lemming.interfaces.Frame;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
@SuppressWarnings("rawtypes")
public class ScifioLoaderTest {

	
	int f;
	ScifioLoader tif;
	QueueStore<Frame> frames;
	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ScifioLoader(p.getProperty("samples.dir")+"eye.tif");
		frames = new QueueStore<Frame>();
		
		tif.setOutput(frames);
	}

	@Test
	public void test() {
		tif.run();
		
		f = frames.getLength();
		
		assertEquals(25, f);		
		
		tif.show();

		LemMING.pause(5000);
	}

}
