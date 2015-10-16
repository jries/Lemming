package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.DoGFinder;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class DoGFinderTest {

	private Manager pipe;
	private ImageLoader tif;
	private DoGFinder peak;
	private SaveLocalizations saver;
	private UnpackElements unpacker;
	private Map<Integer, Store> map;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		//tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif = new ImageLoader(new ImagePlus("/Users/ronny/ownCloud/storm/TubulinAF647.tif"));
		pipe.add(tif);
		
		peak = new DoGFinder(6,100);
		pipe.add(peak);
		
		unpacker = new UnpackElements();
		pipe.add(unpacker);
		
		//saver = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver = new SaveLocalizations(new File("/Users/ronny/ownCloud/storm/dogfinder.csv"));
		pipe.add(saver);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(peak, unpacker);
		pipe.linkModules(unpacker, saver);
		map = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("");		
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
