package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.HashWorkspace;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.inputs.TIFFLoader;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.PeakFinder;
import org.lemming.utils.LemMING;

/**
 * Test class for finding peaks based on a threshold value and inserts the
 * the frame number and the x,y coordinates of the localization into a HashWorkspace.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class PeakFinderWorkspaceTest {
	
	TIFFLoader tif;
	Store<Frame> frames;
	Store<Localization> localizations;
	HashWorkspace h;
	PeakFinder peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new TIFFLoader(p.getProperty("samples.dir")+"eye.tif");
		peak = new PeakFinder(200);
		frames = new QueueStore<Frame>();
		h = new HashWorkspace();
		localizations = h.getFIFO();
		print = new PrintToScreen();
		
		tif.setOutput(frames);
		peak.setInput(frames);
		peak.setOutput(localizations); //this is not used, but needs to be set to not get a NullStoreWarning
		print.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(tif).start();
		new Thread(peak).start();
		new Thread(print).start();
		
		LemMING.pause(3000);
		
		equals(frames.isEmpty());
		
		assertEquals(h.getNumberOfRows(), 324);
		
		System.out.println(h.toString());
		
		assertTrue(h.hasMember(h.getFrameName()));
	}

}
