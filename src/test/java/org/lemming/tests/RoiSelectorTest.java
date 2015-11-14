package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ReadLocalizations;
import org.lemming.modules.RoiSelector;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Manager;
	
@SuppressWarnings("rawtypes")
public class RoiSelectorTest {

	private Manager pipe;
	private ReadLocalizations reader;
	private FastStore locs;
	private RoiSelector selector;
	private FastStore selocs;

	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		locs = new FastStore();
		reader = new ReadLocalizations(new File("/home/ronny/Bilder/out.csv"),",");
		reader.setOutput(locs);
		pipe.add(reader);
		
		selocs = new FastStore();
		selector = new RoiSelector(10, 10, 20, 20,0);
		selector.setInput(locs);
		selector.setOutput(selocs);
		pipe.add(selector);
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("locs:" + locs.getLength() +", selected:" + selocs.getLength());		
		assertEquals(true,locs.isEmpty());
	}

}
