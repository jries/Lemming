package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.queue.StoreSplitter;
import org.lemming.queue.SO;
import org.lemming.queue.SI;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

/**
 * Test class for splitting a Store into multiple Stores.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class StoreSplitterTest {

	Runnable rl;
	Runnable gro;
	Runnable pts;
	QueueStore<Localization> localizations;
	QueueStore<Localization> printLocalizations;
	QueueStore<Localization> renderLocalizations;
	StoreSplitter<Localization> splitter;

	@Before
	public void setUp() throws Exception {
		splitter = new StoreSplitter<Localization>();

		printLocalizations = new QueueStore<Localization>();
		renderLocalizations = new QueueStore<Localization>();
		localizations = new QueueStore<Localization>();
		
                rl = new SO<Localization>(new RandomLocalizer(20, 256, 256), localizations);
                gro = new SI<Localization>(renderLocalizations, new GaussRenderOutput(256, 256));
                pts = new SI<Localization>(printLocalizations, new PrintToScreen());
		
		splitter.setInput(localizations);		
		splitter.addOutput(printLocalizations);
		splitter.addOutput(renderLocalizations);		
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
