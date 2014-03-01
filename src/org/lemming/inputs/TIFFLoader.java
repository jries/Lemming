package org.lemming.inputs;

import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;

import java.util.ArrayList;
import java.util.List;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Store;
import org.lemming.utils.LArrays;
import org.lemming.utils.LFile;

/**
 * Loads a TIFF file and generates objects of type ImgLib2Frame.
 * 
 * @author Joe Borbely, Stephan Preibish, Thomas Pengo
 *
 */
public class TIFFLoader<T extends RealType<T> & NativeType<T>> extends SO<ImgLib2Frame<T>> {

	Img<T> theImage;
	
	Store<ImgLib2Frame<T>> s; 
	LFile file; 
	
	public TIFFLoader(String filename) {
		this(filename, null);
	}
 	
    @SuppressWarnings("unchecked") 
	public TIFFLoader(String filename, T type) {
    	// create the filename reference
    	file = new LFile(filename);

    	// check that the filename is a TIFF file
    	List<String> extns = new ArrayList<String>(LArrays.asList("tiff", "tif"));
    	String ext = file.getExtension();
    	if (!extns.contains(ext))
    		throw new RuntimeException("TIFFLoader cannot open files with a ."+ext+" extension");

    	// make sure that the TIFF file exists
    	if (!file.exists())
    		throw new RuntimeException("A '" + filename + "' file does not exist");    
    	
    	try {
    		if (type == null )
    			theImage = (Img<T>) (Object) new ImgOpener().openImg( file.getAbsolutePath(), new ArrayImgFactory<>());
    		else
    			theImage = new ImgOpener().openImg( file.getAbsolutePath(), new ArrayImgFactory<T>(), type );
		} catch (ImgIOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
}

    long curSlice;
    
    @Override
	public void beforeRun() {
		curSlice = 0;
 	}
    
	@Override
	public boolean hasMoreOutputs() {
		return curSlice < theImage.dimension(2);
	}

	@Override
	public ImgLib2Frame<T> newOutput() {
		ImgLib2Frame<T> out = new ImgLib2Frame<T>(curSlice, (int)theImage.dimension(0), (int)theImage.dimension(1), Views.hyperSlice(theImage, 2, curSlice)); 
		curSlice++;
		return out;
	}
	
	/** Display the TIFF file using ImageJ */
	public void show() {
		ImageJFunctions.show( theImage );		
	}

}
