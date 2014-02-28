package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.NonblockingQueueStore;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.Rendering;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.HistogramRender;
import org.lemming.utils.LemMING;

/**
 * Test class for creating randomly-positioned localizations and 
 * adding the localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class RandomLocalizerTest {

	RandomLocalizer rl;
	Rendering ren;
	Store<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {
		//localizations = new NonblockingQueueStore<Localization>();
		localizations = new QueueStore<Localization>();
		
		rl = new RandomLocalizer(50000, 256, 256);
		rl.setOutput(localizations);
	}

	@Test
	public void testGaussRender() {
		ren = new GaussRenderOutput(256, 256);
		ren.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(ren).start();

		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}
		
		LemMING.pause(2000);
	}

	@Test
	public void testHistoRender() {
		ren = new HistogramRender();
		ren.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(ren).start();
		
		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}
		
		LemMING.pause(2000);
	}

	@Test
	public void testMultiplHistoRender() {
		ren = new HistogramRender();
		ren.setInput(localizations);
		
		Rendering histo2 = new HistogramRender();
		histo2.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(ren).start();
		new Thread(histo2).start();
		
		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}
		
		LemMING.pause(2000);
	}
	
	/*
	@Test
	public void testScatterPlot() {
		ren = new Jzy3dScatterplot();
		ren.setInput(localizations);
		
		new Thread(rl).start();
		
		ren.run();
		
		LemMING.pause(10000);
	}
	*/

}
