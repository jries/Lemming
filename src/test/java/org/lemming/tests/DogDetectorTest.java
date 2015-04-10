package org.lemming.tests;

import java.io.File;

import net.imglib2.util.Util;
import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;
import org.lemming.outputs.PrintToFile;
import org.lemming.processors.DogDetector;

@SuppressWarnings("rawtypes")
public class DogDetectorTest {

	private ImageJTIFFLoader tif;
	private DogDetector detector;
	private double[] calibration;
	private QueueStore<Frame> frames;
	private QueueStore<Localization> localizations;
	private PrintToFile print;
	

	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		tif = new ImageJTIFFLoader("/media/data/temporary/storm/TubulinAF647.tif");
		calibration = Util.getArrayFromValue( 1d, 2 );
		detector = new DogDetector(6,calibration,330);
		frames = new QueueStore<Frame>();
		localizations = new QueueStore<Localization>();
		File f = new File("/media/data/temporary/storm/results.csv");
		print = new PrintToFile(f);
		
		tif.setOutput(frames);
		detector.setInput(frames);
		detector.setOutput(localizations);
		print.setInput(localizations);
	}

	@Test
	public void test() {
		long startTime = System.currentTimeMillis();
		Thread t_load = new Thread(tif,"ImageJTIFFLoader");
		Thread t_detector = new Thread(detector,"DogDetector");
		Thread t_print = new Thread(print,"PrintToFile");
		
		t_load.start();
		/*if (frames.isEmpty())
			try {
				Thread.sleep(10);
				System.out.println("delay of 10 ms");
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}*/
		t_detector.start();
		t_print.start();
		
		try {
			t_load.join();
			t_detector.join();
			t_print.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
		long endTime = System.currentTimeMillis();
		System.out.println(endTime-startTime);
		assertEquals(true,frames.isEmpty());
	}


}
