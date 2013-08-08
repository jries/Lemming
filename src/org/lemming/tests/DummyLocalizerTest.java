package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.interfaces.Input;

public class DummyLocalizerTest {

	DummyLocalizer d;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyLocalizer();
	}

	@Test
	public void test() {
		QueueStore<Frame> frames = new QueueStore<>();
		QueueStore<Localization> localizations = new QueueStore<Localization>();
		
		DummyFrameProducer i = new DummyFrameProducer();
		
		i.setOutput(frames);
		d.setInput(frames);
		d.setOutput(localizations);
		
		new Thread(i).start();
		new Thread(d).start();
		
		try {
			Thread.sleep(1000);
			
			assertEquals(localizations.getLength(), 200);
			assertEquals(frames.getLength(), 0);
			
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

}
