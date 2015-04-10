package org.lemming.utils;

import java.io.File;
import java.net.URI;

/**This Lemming-File (LFile) wrapper class is a representation of file and 
 * directory pathnames that implements all of the {@link java.io.File} methods,
 * plus custom methods */
public class LFile extends File {

    private static final long serialVersionUID = 1L;
    
    /** Creates a new File instance from a parent abstract pathname and a child pathname string. 
     * @see java.io.File#File(File parent, String child) */
    public LFile(File parent, String child) {
        super(parent, child);
    }
    
    /** Creates a new File instance by converting the given pathname string into an abstract pathname.
     * @see java.io.File#File(String pathname) */
    public LFile(String pathname) {
        super(pathname);
    }

    /**Creates a new File instance from a parent pathname string and a child pathname string.
     * @see java.io.File#File(String parent, String child) */
    public LFile(String parent, String child) {
        super(parent, child);
    }    
    
    /** Creates a new File instance by converting the given file: URI into an abstract pathname.
     * @see java.io.File#File(URI uri) */
    public LFile(URI uri) {
        super(uri);
    }

    
    /*
     * 
     * Start of custom methods
     * 
     */    
    
	/** Returns the file extension. If there is no file extension then an empty string is returned. 
	 * @return file extension*/
    public String getExtension() {
		String s = super.getName();
		int i = s.lastIndexOf('.');
		if (i > 0 && i < s.length() - 1)
			return s.substring(i+1).toLowerCase();
		return "";
    }
        
}
