package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.utils.LemMING;

/**
 * Test class for inserting dummy frames into a Store and dummy 
 * localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class DummyLocalizerTest {

	DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d;
	QueueStore<ImgLib2Frame<UnsignedShortType>> frames;
	QueueStore<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyLocalizer();
		frames = new QueueStore();
		localizations = new QueueStore<Localization>();
	}

	@Test
	public void test() {
		DummyFrameProducer i = new DummyFrameProducer();
		
		i.setOutput(frames);
		d.setInput(frames);
		d.setOutput(localizations);
		
		new Thread(i).start();
		new Thread(d).start();
		
		LemMING.pause(1000);
		
		assertEquals(localizations.getLength(), 200);
		assertEquals(frames.getLength(), 0);
	}

}
