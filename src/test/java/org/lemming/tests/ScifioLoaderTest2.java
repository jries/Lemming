package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.inputs.ScifioLoader;
import org.lemming.interfaces.Frame;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
@SuppressWarnings("rawtypes")
public class ScifioLoaderTest2 {

	ScifioLoader tif;
	ImageJTIFFLoader ij;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		
		tif = new ScifioLoader("/media/data/temporary/storm/TubulinAF647.tif");
		ij = new ImageJTIFFLoader("/media/data/temporary/storm/TubulinAF647.tif");
		frames = new QueueStore<Frame>();
	}

	@SuppressWarnings("unchecked")
	@Test
	public void test() {
		long startTime = System.currentTimeMillis();
		tif.setOutput(frames);
		
		
		Thread t_i = new Thread(tif,"SCIFIOImport");
		t_i.start();
		
		try {
			t_i.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
		
		long endTime = System.currentTimeMillis();
		System.out.println(endTime-startTime);
		
		startTime = System.currentTimeMillis();
		ij.setOutput(frames);
		
		Thread t_j = new Thread(ij,"IJImport");
		t_j.start();
		
		try {
			t_j.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
		
		endTime = System.currentTimeMillis();
		System.out.println(endTime-startTime);
		
		assertEquals(9990, frames.getLength());		
		
		ij.show();

		LemMING.pause(5000);
	}

}
