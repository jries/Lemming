package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.StoreLoader;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.HistogramRenderer;

public class HistogramRendererTest {

	private Manager pipe;
	private StoreLoader reader;
	private HistogramRenderer histo;

	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		reader = new StoreLoader(new File(System.getProperty("user.home")+"/ownCloud/storm/geomTable.csv"),",");
		pipe.add(reader);
		
		histo = new HistogramRenderer(1024, 1024, -3, 10, -3, 10, 0, 100);
		pipe.add(histo);
		pipe.linkModules(reader, histo, true, 100);
		
	}

	@Test
	public void test() {
		pipe.run();
		histo.show();
		assertEquals(true,pipe.getMap().values().iterator().next().isEmpty());
	}

}
