package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;
import java.util.concurrent.Executors;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ReadLocalizationPrecision3D;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.GaussRenderer;

public class GaussRendererTest {

	private Manager pipe;
	private GaussRenderer gauss;
	private Map<Integer, Store> map;

	@Before
	public void setUp() {
		pipe = new Manager(Executors.newCachedThreadPool());

		ReadLocalizationPrecision3D reader = new ReadLocalizationPrecision3D(new File(System.getProperty("user.home") + "/ownCloud/storm/fitted.csv"), ",");
		pipe.add(reader);
		
		gauss = new GaussRenderer(512, 512, 22.200, 22.800, 27.000, 27.600);
		pipe.add(gauss);
		
		pipe.linkModules(reader, gauss, true, 100);
		map = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		gauss.show();
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
