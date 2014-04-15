package org.lemming.tests;

import java.io.FileReader;
import java.util.AbstractList;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.queue.Store;
import org.lemming.inputs.ScifioLoader;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.PeakFinder;
import org.lemming.utils.LemMING;

/**
 * Test class for finding peaks based on a threshold value and inserts the
 * the frame number and the x,y coordinates of the localization into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class PeakFinderTest {
	
	ScifioLoader tif;
	PeakFinder peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ScifioLoader(p.getProperty("samples.dir")+"eye.tif");
		peak = new PeakFinder(200);
		print = new PrintToScreen();
	}

	@Test
	public void test() {
                while (tif.hasMoreOutputs()) {
                        AbstractList<Localization> peaks = peak.process(tif.newOutput());
                        for (Localization localization : peaks) {
                                print.process(localization);
                        }
                }
	}

}
