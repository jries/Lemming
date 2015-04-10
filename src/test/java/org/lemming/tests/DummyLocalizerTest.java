package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.FastStore;
import org.lemming.data.ImgLib2Frame;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.interfaces.Localization;

/**
 * Test class for inserting dummy frames into a Store and dummy 
 * localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo, Ronny Sczech
 */
public class DummyLocalizerTest {

	private DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d;
	private FastStore<ImgLib2Frame<UnsignedShortType>> frames;
	private FastStore<Localization> localizations;
	private DummyFrameProducer i;
	
	@Before
	public void setUp() throws Exception {
		i = new DummyFrameProducer();
		d = new DummyLocalizer<>();
		frames = new FastStore<ImgLib2Frame<UnsignedShortType>>();
		localizations = new FastStore<Localization>();
	}

	@Test
	public void test() {
		long startTime = System.currentTimeMillis();
		i.setOutput(frames);
		d.setInput(frames);
		d.setOutput(localizations);
		Thread t_i = new Thread(i,"DummyFrameProducer");
		Thread t_d = new Thread(d,"DummyLocalizer");
		t_i.start();
		t_d.start();
		try {
			t_i.join();
			t_d.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
		
		long endTime = System.currentTimeMillis();
		System.out.println(endTime-startTime);
		assertEquals(localizations.getLength(), 200);
		assertEquals(frames.getLength(), 0);
	}
}
