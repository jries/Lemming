package org.lemming.tests;

import static org.junit.Assert.*;
import ij.ImageJ;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.input.RandomLocalizer;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.StoreSplitter;

public class StoreSplitterTest {

	RandomLocalizer rl;
	GaussRenderOutput gro;
	PrintToScreen pts;
	QueueStore<Localization> localizations;
	QueueStore<Localization> printLocalizations;
	QueueStore<Localization> renderLocalizations;
	StoreSplitter<Localization> splitter;

	@Before
	public void setUp() throws Exception {

		String[] arg = {""};
		ImageJ.main(arg);
		
		rl = new RandomLocalizer(20, 256, 256);
		gro = new GaussRenderOutput(256, 256);
		pts = new PrintToScreen();
		splitter = new StoreSplitter<Localization>();
		
		printLocalizations = new QueueStore<Localization>();
		renderLocalizations = new QueueStore<Localization>();
		localizations = new QueueStore<Localization>();
		
		rl.setOutput(localizations);
		splitter.setInput(localizations);		
		splitter.addOutput(printLocalizations);
		splitter.addOutput(renderLocalizations);		
		gro.setInput(renderLocalizations);
		pts.setInput(printLocalizations);
	}

	@Test
	public void test() {
		new Thread(rl).start();
		new Thread(splitter).start();
		new Thread(gro).start();
		new Thread(pts).start();
		
		try {
			Thread.sleep(100);			
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		assertEquals(printLocalizations.getLength(), 0);
		assertEquals(renderLocalizations.getLength(), 0);
		
		while (true){}		
	}

}
