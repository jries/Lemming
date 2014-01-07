package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.inputs.TIFFLoader;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.PeakFinder;
import org.lemming.processors.WindowPeakFinder;

public class WindowPeakFinderTest {
	
	TIFFLoader tif;
	Store<Frame> frames;
	Store<Localization> localizations;
	WindowPeakFinder peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new TIFFLoader(p.getProperty("samples.dir")+"eye.tif");
		peak = new WindowPeakFinder(200);
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
