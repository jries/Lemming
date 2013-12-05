package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.NonblockingQueueStore;
import org.lemming.data.QueueStore;
import org.lemming.data.Rendering;
import org.lemming.data.Store;
import org.lemming.input.RandomLocalizer;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.HistogramRender;
import org.lemming.outputs.Jzy3dScatterplot;

public class RandomLocalizerTest {

	RandomLocalizer rl;
	Rendering gro;
	Store<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {
		localizations = new NonblockingQueueStore<Localization>();
		
		rl = new RandomLocalizer(50000, 256, 256);
		rl.setOutput(localizations);
	}

	@Test
	public void test1() {
		gro = new GaussRenderOutput(256, 256);
		gro.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(gro).start();

		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				break;
			}
		}		
		
	}

	@Test
	public void test2() {
		gro = new HistogramRender(1024,1024,0,255,0,255);
		gro.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(gro).start();

		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				return;
			}
		}		
	}

	@Test
	public void test3() {
		gro = new HistogramRender(1024,1024,0,255,0,255);
		gro.setInput(localizations);
		
		Rendering bro = new HistogramRender(1024,1024,0,255,0,255);
		bro.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(gro).start();
		new Thread(bro).start();
		
		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				return;
			}
		}		
	}
	
	@Test
	public void test4() {
		gro = new Jzy3dScatterplot();
		gro.setInput(localizations);
		
		new Thread(rl).start();
		
		gro.run();
		
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			return;
		}

	}

}
