package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizationPrecision3D;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.Manager;
import org.lemming.pipeline.Settings;
import org.lemming.plugins.AstigFitter;
import org.lemming.plugins.NMSDetector;

import ij.ImagePlus;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

@SuppressWarnings("rawtypes")
public class AstigFitterTest {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	private ImagePlus loc_im;
	
	@Before
	public void setUp() throws Exception {
		
        File file = new File(System.getProperty("user.home")+"/ownCloud/storm/p500ast_.tif");
        
		if (file.isDirectory()){
        	FolderOpener fo = new FolderOpener();
        	fo.openAsVirtualStack(true);
        	loc_im = fo.openFolder(file.getAbsolutePath());
        }
        
        if (file.isFile()){
        	loc_im = FileInfoVirtualStack.openVirtual(file.getAbsolutePath());
        }
	
	    if (loc_im ==null)
		    throw new Exception("File not found");
		
		AbstractModule tif = new ImageLoader(loc_im);

		AbstractModule peak = new NMSDetector(700,7);
		AbstractModule fitter = new AstigFitter<>(7, Settings.readCSV(System.getProperty("user.home")+"/ownCloud/storm/calTest.csv").get("param"));
		AbstractModule saver = new SaveLocalizationPrecision3D(new File(System.getProperty("user.home")+"/ownCloud/storm/test3.csv"));
		
		pipe = new Manager();
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(saver);
		
		pipe.linkModules(tif, peak, true, loc_im.getStackSize());
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver,false, 100);
		storeMap = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
