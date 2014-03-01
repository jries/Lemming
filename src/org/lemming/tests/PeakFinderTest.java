package org.lemming.tests;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
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
	Store<Frame> frames;
	Store<Localization> localizations;
	PeakFinder peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ScifioLoader(p.getProperty("samples.dir")+"eye.tif");
		peak = new PeakFinder(200);
		frames = new QueueStore<Frame>();
		localizations = new QueueStore<Localization>();
		print = new PrintToScreen();
		
		tif.setOutput(frames);
		peak.setInput(frames);
		peak.setOutput(localizations); //this is not used, but needs to be set in order to not get a NullStoreWarning
		print.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(tif).start();
		new Thread(peak).start();
		new Thread(print).start();
		
		LemMING.pause(2000);
		
		equals(frames.isEmpty());
	}

}
