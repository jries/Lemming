package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ReadFittedLocalizations;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.GaussRenderer;

public class GaussRendererTest {

	private Manager pipe;
	private ReadFittedLocalizations reader;
	private GaussRenderer gauss;
	private Map<Integer, Store> map;

	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		reader = new ReadFittedLocalizations(new File(System.getProperty("user.home")+"/ownCloud/storm/fitted.csv"),",");
		pipe.add(reader);
		
		gauss = new GaussRenderer(22.200, 22.800, 27.000, 27.600, 0.00200, 0.00200);
		pipe.add(gauss);
		
		pipe.linkModules(reader, gauss, true);
		map = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
