package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.PeakFinder;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.input.TIFFLoader;
import org.lemming.outputs.PrintToScreen;

public class PeakFinderTest {
	
	TIFFLoader tif;
	Store<Frame> frames;
	Store<Localization> localizations;
	PeakFinder peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		tif = new TIFFLoader("D:/microscopy_software/Localization Microscopy Challenge/Eye/eye.tif");
		peak = new PeakFinder(200);
		frames = new QueueStore<Frame>();
		localizations = new QueueStore<Localization>();
		print = new PrintToScreen();
		
		tif.setOutput(frames);
		peak.setInput(frames);
		peak.setOutput(localizations); //this is not used yet
		print.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(tif).start();
		new Thread(peak).start();
		new Thread(print).start();
		
		try {
			Thread.sleep(2000);
			
			equals(frames.isEmpty());
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	}

}
