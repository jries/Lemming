package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.StoreSplitter;
import org.lemming.utils.LemMING;

/**
 * Test class for splitting a Store into multiple Stores.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
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
		rl = new RandomLocalizer(20, 256, 256);
		gro = new GaussRenderOutput(256, 256);
		gro.setTitle("Store Splitter Test");
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
		
		LemMING.pause(1000);

		assertEquals(printLocalizations.getLength(), 0);
		assertEquals(renderLocalizations.getLength(), 0);
		
		LemMING.pause(2000);
	}

}
