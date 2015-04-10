package org.lemming.tests;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.PeakFinder;
import org.lemming.utils.LemMING;

/**
 * Test class for finding peaks based on a threshold value and inserts the
 * the frame number and the x,y coordinates of the localization into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
@SuppressWarnings("rawtypes")
public class PeakFinderTest {
	
	ImageJTIFFLoader tif;
	Store<Frame> frames;
	Store<Localization> localizations;
	PeakFinder peak;
	PrintToScreen print;

	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ImageJTIFFLoader(p.getProperty("samples.dir")+"4b_green.tif");
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
