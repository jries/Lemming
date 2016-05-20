package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;
import java.util.concurrent.Executors;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.NMSDetector;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class NMSFinderTest {

	private Manager pipe;
	private Map<Integer, Store> map;
	
	@Before
	public void setUp() {
		pipe = new Manager(Executors.newCachedThreadPool());
		final ImagePlus image = new ImagePlus(System.getProperty("user.home")+"/ownCloud/p500ast_.tif");
		ImageLoader tif = new ImageLoader<>(image, LemmingUtils.readCameraSettings("camera.props"));
		pipe.add(tif);

		NMSDetector peak = new NMSDetector(700, 9, 0);
		pipe.add(peak);

		UnpackElements unpacker = new UnpackElements();
		pipe.add(unpacker);

		SaveLocalizations saver = new SaveLocalizations(new File(System.getProperty("user.home") + "/ownCloud/nmsfinder.csv"));
		pipe.add(saver);
		
		pipe.linkModules(tif, peak, true, image.getStackSize());
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
