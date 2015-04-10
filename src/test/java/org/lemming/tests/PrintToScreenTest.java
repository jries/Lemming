package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a Store of localizations and then writing the 
 * localizations to the screen.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class PrintToScreenTest {

	PrintToScreen p;
	
	@Before
	public void setUp() throws Exception {
		p = new PrintToScreen();
	}

	@Test
	public void test() {
		QueueStore<ImgLib2Frame<UnsignedShortType>> frames = new QueueStore<>();
		QueueStore<Localization> localizations = new QueueStore<Localization>();
		
		DummyFrameProducer i = new DummyFrameProducer();
		DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d1 = new DummyLocalizer<>();
		DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d2 = new DummyLocalizer<>();
		
		i.setOutput(frames);
		d1.setInput(frames);
		d1.setOutput(localizations);
		d2.setInput(frames);
		d2.setOutput(localizations);
		p.setInput(localizations);
		
		new Thread(i).start();
		new Thread(d1).start();
		new Thread(d2).start();
		new Thread(p).start();
		
		LemMING.pause(1000);
		
		assertEquals(localizations.getLength(), 0);
		assertEquals(frames.getLength(), 0);
	}

}
