package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
//import org.lemming.modules.ImageMath;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.Manager;
//import org.lemming.plugins.DoGFinder;
//import org.lemming.plugins.FastMedianFilter;
import org.lemming.plugins.MLE_Fitter;
import org.lemming.plugins.NMSDetector;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

@SuppressWarnings("rawtypes")
public class GPUFitterTest {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	private ImagePlus loc_im;
	
	
	@Before
	public void setUp() throws Exception {
		
        //File file = new File("D:/Images/DRG_KO_5_1.tif");
		//File file = new File("D:/Images/DRG_WT_MT_A647_1.tif");
        File file = new File("/media/backup/ownCloud/Tubulin1-1.tif");
		//File file = new File("D:/ownCloud/Tubulin1.tif");
        
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
	    
		AbstractModule tif = new ImageLoader<>(loc_im, LemmingUtils.readCameraSettings("camera.props"));
		//AbstractModule filter = new FastMedianFilter(3, false);
		//ImageMath substracter = new ImageMath(3);
		//substracter.setOperator(ImageMath.operators.SUBSTRACTION);
		//AbstractModule peak = new NMSDetector(70,7);
		AbstractModule peak = new NMSDetector(350,6,10); //
		//AbstractModule peak = new DoGFinder(4.5f,13); //DRG_KO_5_1.tif
		//AbstractModule peak = new NMSDetector(2000,5); //DRG_WT_MT_A647_1.tif
		AbstractModule fitter = new MLE_Fitter<>(6);
		AbstractModule saver = new SaveLocalizations(new File("/media/backup/ownCloud/Tubulin1-1.csv"));
		AbstractModule unpacker = new UnpackElements();
		AbstractModule saver2 = new SaveLocalizations(new File("/media/backup/ownCloud/Tubulin1-1det.csv"));
		
		pipe = new Manager();
		pipe.add(tif);
		//pipe.add(substracter);
		//pipe.add(filter);
		pipe.add(peak);
		pipe.add(unpacker);
		pipe.add(fitter);
		pipe.add(saver);
		pipe.add(saver2);
		
		pipe.linkModules(tif, peak, true, loc_im.getStackSize());
		pipe.linkModules(peak, unpacker);
		pipe.linkModules(unpacker, saver2);
		//pipe.linkModules(tif, substracter);
		//pipe.linkModules(filter, substracter);
		//pipe.linkModules(substracter, peak);
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver,false, 128);
		storeMap = pipe.getMap();
	}

	@Test
	public void test() {
		long start = System.currentTimeMillis();
		pipe.run();
		long end = System.currentTimeMillis();
		System.out.println("Overall: " + (end-start) +"ms");
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
